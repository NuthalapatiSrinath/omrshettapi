// server.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
dotenv.config(); // load .env from project root

import { getLocalIP } from './utils/getLocalIp.js';
import { sendResponse } from './services/responseService.js';
import connectDB from './database/connectDB.js';

import omrRoutes from './routes/omr/omr.routes.js';
import omrReaderRoutes from './routes/omrReader/omrReader.routes.js';
import errorHandler from './services/errorHandler.js';

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

/**
 * Robust parsing for allowed origins/methods environment vars.
 * Falls back to sensible defaults if env vars are missing or empty.
 */
const DEFAULT_ORIGINS = ['http://localhost:3000', 'http://localhost:5173'];
const DEFAULT_METHODS = ['GET', 'POST', 'PATCH', 'OPTIONS'];

const allowedOrigins = (process.env.ALLOWED_ORIGINS?.trim()
  ? process.env.ALLOWED_ORIGINS.split(',').map(s => s.trim())
  : DEFAULT_ORIGINS);

const allowedMethods = (process.env.ALLOWED_METHODS?.trim()
  ? process.env.ALLOWED_METHODS.split(',').map(s => s.trim().toUpperCase())
  : DEFAULT_METHODS);

// helpful startup logging for debugging
console.log('Starting server with these env flags (presence only):', {
  NODE_ENV: process.env.NODE_ENV || 'development',
  PORT: process.env.PORT || 'not set (will default to 3000)',
  MONGO_URI: !!process.env.MONGO_URI,
  S3_BUCKET_NAME: !!process.env.S3_BUCKET_NAME,
  S3_BUCKET_REGION: !!process.env.S3_BUCKET_REGION,
  S3_ACCESS_KEY: !!process.env.S3_ACCESS_KEY,
  SECRET_ACCESS_KEY: !!process.env.SECRET_ACCESS_KEY,
});
console.log('CORS allowed origins:', allowedOrigins);
console.log('CORS allowed methods:', allowedMethods);

// CORS configuration (defensive)
app.use(
  cors({
    origin: function (origin, callback) {
      // allow requests with no origin (mobile apps, curl, some tools)
      if (!origin) return callback(null, true);
      if (allowedOrigins.indexOf(origin) !== -1) {
        return callback(null, true);
      } else {
        return callback(new Error('Not allowed by CORS'), false);
      }
    },
    methods: allowedMethods,
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
  })
);

// Routes
app.use('/omr', omrRoutes);
app.use('/omrReader', omrReaderRoutes);

// Basic root
app.get('/', (req, res) => {
  sendResponse(res, 200, 'Welcome to the API', null, null);
});

// Health check
app.get('/health', (req, res) => {
  sendResponse(res, 200, 'Server is running', null, null);
});

// Catch-all for undefined endpoints
app.use((req, res) => {
  sendResponse(res, 404, 'Route Not Found', null, {
    message: `The route ${req.originalUrl} does not exist on this server.`,
  });
});

// Error handler (centralized)
app.use(errorHandler);

// Ensure PORT is numeric and provide default
const PORT = Number(process.env.PORT) || 3000;

// Bind to all interfaces (0.0.0.0) so LAN IP (e.g., 192.168.x.x) can reach it
app.listen(PORT, '0.0.0.0', async () => {
  try {
    await connectDB();
    const ip = getLocalIP();
    console.log('Server is running at :');
    console.log(`http://${ip}:${PORT}  (LAN address)`);
    console.log(`http://localhost:${PORT}  (localhost)`);
    console.log('CORS allowed origins (runtime):', allowedOrigins);
  } catch (error) {
    console.error(`Database connection error: ${error?.message || error}`);
    process.exit(1);
  }
});
