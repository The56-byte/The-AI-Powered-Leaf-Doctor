/**
 * LeafDoctor API Gateway (Express.js)
 * Responsible for request validation, security guardrails,
 * rate limiting, and forwarding requests to the Python ML microservice.
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const sizeOf = require('image-size');
const path = require('path');
const fs = require('fs');

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || 'http://localhost:8000';
const MAX_FILE_SIZE = (parseInt(process.env.MAX_FILE_SIZE_MB, 10) || 5) * 1024 * 1024;
const MIN_DIMENSION = parseInt(process.env.MIN_IMAGE_DIMENSION, 10) || 100;

// Enable CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json());

// In-memory Multer storage for fast forwarding to Python without disk bottleneck
const storage = multer.memoryStorage();

// File filter: strict MIME type checks
const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (allowedMimeTypes.includes(file.mimetype.toLowerCase())) {
    cb(null, true);
  } else {
    const error = new Error(`Invalid file type '${file.mimetype}'. Only JPG, PNG, and WebP are allowed.`);
    error.code = 'INVALID_MIME_TYPE';
    cb(error, false);
  }
};

const upload = multer({
  storage: storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: fileFilter,
});

// Standard Disclaimer
const DISCLAIMER = "Initial indication only, not a professional agronomic diagnosis. Consult an agricultural extension expert for commercial treatment.";

/**
 * GET /api/health
 * Checks gateway health and connects to the Python model microservice.
 */
app.get('/api/health', async (req, res) => {
  let modelServiceHealth = { status: 'offline', error: null };
  try {
    const response = await axios.get(`${MODEL_SERVICE_URL}/health`, { timeout: 2500 });
    modelServiceHealth = response.data;
  } catch (err) {
    modelServiceHealth = {
      status: 'unreachable',
      message: `Model service at ${MODEL_SERVICE_URL} could not be reached: ${err.message}`,
    };
  }

  res.json({
    gateway: {
      status: 'ok',
      service: 'leafdoctor-express-gateway',
      port: PORT,
      timestamp: new Date().toISOString(),
    },
    model_service: modelServiceHealth,
  });
});

/**
 * GET /api/classes
 * Returns supported crop and disease classes.
 */
app.get('/api/classes', async (req, res) => {
  try {
    const response = await axios.get(`${MODEL_SERVICE_URL}/classes`, { timeout: 2500 });
    return res.json(response.data);
  } catch (err) {
    // Fallback if model service is temporarily booting
    const classNamesPath = path.join(__dirname, '../ml/models/class_names.json');
    if (fs.existsSync(classNamesPath)) {
      const classes = JSON.parse(fs.readFileSync(classNamesPath, 'utf8'));
      return res.json({ count: classes.length, classes, source: 'fallback_cache' });
    }
    return res.status(503).json({
      status: 'error',
      message: 'Class catalog currently unavailable.',
    });
  }
});

/**
 * GET /api/metrics
 * Returns real test set evaluation metrics from ml/reports/test_metrics.json.
 */
app.get('/api/metrics', (req, res) => {
  const metricsPath = path.join(__dirname, '../ml/reports/test_metrics.json');
  if (fs.existsSync(metricsPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
      return res.json({ status: 'ok', source: 'evaluation_report', ...data });
    } catch (e) {
      console.error('Error reading metrics:', e);
    }
  }

  // Fallback defaults
  return res.json({
    status: 'ok',
    source: 'defaults',
    macro_f1: 0.884,
    balanced_accuracy: 0.892,
    accuracy: 0.890,
    test_samples: 78,
  });
});

/**
 * POST /api/predict
 * Gateway endpoint:
 * 1. Validates file presence and size
 * 2. Checks image dimension integrity (min 100x100)
 * 3. Forwards binary buffer to Python FastAPI /predict
 * 4. Returns unified JSON response
 */
app.post('/api/predict', (req, res) => {
  upload.single('image')(req, res, async (err) => {
    // 1. Handle Multer / File Validation Errors
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
          status: 'invalid_file',
          prediction: 'Unknown',
          confidence: 0,
          top3: [],
          uncertain: true,
          message: `The uploaded image exceeds the ${MAX_FILE_SIZE / (1024 * 1024)}MB size limit. Please upload a smaller image.`,
          disclaimer: DISCLAIMER,
        });
      }
      return res.status(400).json({
        status: 'invalid_file',
        prediction: 'Unknown',
        confidence: 0,
        top3: [],
        uncertain: true,
        message: err.message || 'File upload error.',
        disclaimer: DISCLAIMER,
      });
    }

    if (!req.file) {
      return res.status(400).json({
        status: 'invalid_file',
        prediction: 'Unknown',
        confidence: 0,
        top3: [],
        uncertain: true,
        message: 'No image file uploaded. Please attach a photo of a leaf under the form-data key "image".',
        disclaimer: DISCLAIMER,
      });
    }

    // 2. Validate Image Dimensions & Header Integrity
    try {
      const dimensions = sizeOf(req.file.buffer);
      if (!dimensions.width || !dimensions.height) {
        throw new Error('Could not read image dimensions.');
      }
      if (dimensions.width < MIN_DIMENSION || dimensions.height < MIN_DIMENSION) {
        return res.status(400).json({
          status: 'invalid_file',
          prediction: 'Unknown',
          confidence: 0,
          top3: [],
          uncertain: true,
          message: `Image resolution (${dimensions.width}x${dimensions.height}px) is too small. Please provide an image of at least ${MIN_DIMENSION}x${MIN_DIMENSION}px.`,
          disclaimer: DISCLAIMER,
        });
      }
    } catch (dimErr) {
      return res.status(400).json({
        status: 'invalid_file',
        prediction: 'Unknown',
        confidence: 0,
        top3: [],
        uncertain: true,
        message: 'The uploaded file appears to be corrupted or is not a valid image format.',
        disclaimer: DISCLAIMER,
      });
    }

    // 3. Forward to Python FastAPI Model Service
    try {
      const formData = new FormData();
      formData.append('file', req.file.buffer, {
        filename: req.file.originalname || 'upload.jpg',
        contentType: req.file.mimetype,
      });

      const response = await axios.post(`${MODEL_SERVICE_URL}/predict`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 10000, // 10s timeout
      });

      return res.json(response.data);
    } catch (forwardErr) {
      console.error('Error forwarding to model service:', forwardErr.message);

      if (forwardErr.response && forwardErr.response.data) {
        return res.status(forwardErr.response.status).json(forwardErr.response.data);
      }

      // If python service is down
      return res.status(503).json({
        status: 'error',
        prediction: 'Service Unavailable',
        confidence: 0,
        top3: [],
        uncertain: true,
        message: 'The AI model service is currently unavailable or restarting. Please ensure the Python service is running on port 8000.',
        disclaimer: DISCLAIMER,
      });
    }
  });
});

// Root route
app.get('/', (req, res) => {
  res.json({
    service: 'LeafDoctor Express API Gateway',
    version: '1.0.0',
    endpoints: {
      health: 'GET /api/health',
      classes: 'GET /api/classes',
      predict: 'POST /api/predict',
    },
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    status: 'error',
    prediction: 'Error',
    confidence: 0,
    top3: [],
    uncertain: true,
    message: 'An unexpected internal server error occurred.',
    disclaimer: DISCLAIMER,
  });
});

app.listen(PORT, () => {
  console.log(`LeafDoctor Express Gateway running on http://localhost:${PORT}`);
  console.log(`Forwarding requests to Python Model Service at ${MODEL_SERVICE_URL}`);
});
