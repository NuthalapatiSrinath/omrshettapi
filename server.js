// server.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
dotenv.config();

import connectDB from './database/connectDB.js'; // keep your existing connectDB
import { sendResponse } from './services/responseService.js'; // your helper
import errorHandler from './services/errorHandler.js';
import { getLocalIP } from './utils/getLocalIp.js';

import omrRoutes from './routes/omr/omr.routes.js';
import omrReaderRoutes from './routes/omrReader/omrReader.routes.js';

const app = express();

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// --- Safe env parsing ---
const DEFAULT_ORIGINS = ['http://localhost:3000', 'http://localhost:5173'];
const DEFAULT_METHODS = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE', 'OPTIONS'];

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS && process.env.ALLOWED_ORIGINS.trim() !== '')
  ? process.env.ALLOWED_ORIGINS.split(',').map(s => s.trim()).filter(Boolean)
  : DEFAULT_ORIGINS.slice();

const ALLOWED_METHODS = (process.env.ALLOWED_METHODS && process.env.ALLOWED_METHODS.trim() !== '')
  ? process.env.ALLOWED_METHODS.split(',').map(s => s.trim()).filter(Boolean).concat(['OPTIONS'])
  : DEFAULT_METHODS.slice();

// allow ngrok dynamic URLs + optional NGROK_URL
const ngrokRegex = /^https:\/\/[a-z0-9-]+\.ngrok\.io$/i;
if (process.env.NGROK_URL && process.env.NGROK_URL.trim() !== '') {
  ALLOWED_ORIGINS.push(process.env.NGROK_URL.trim());
}

app.use(cors({
  origin: function (origin, callback) {
    if (!origin) return callback(null, true); // allow curl/postman
    if (ALLOWED_ORIGINS.includes(origin) || ngrokRegex.test(origin)) return callback(null, true);
    return callback(new Error(`CORS blocked: ${origin} not allowed`), false);
  },
  methods: ALLOWED_METHODS,
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// --- routes ---
app.use('/omr', omrRoutes);
app.use('/omrReader', omrReaderRoutes);

app.get('/', (req, res) => sendResponse(res, 200, 'Welcome to the API', null, null));
app.get('/health', (req, res) => sendResponse(res, 200, 'Server is running', null, null));

// catch-all for unknown routes
app.use((req, res) => {
  sendResponse(res, 404, 'Route Not Found', null, {
    message: `The route ${req.originalUrl} does not exist on this server.`
  });
});

// error handler
app.use(errorHandler);

// --- start server ---
const PORT = Number(process.env.PORT) || 5000;
app.listen(PORT, async () => {
  try {
    await connectDB();
    const ip = getLocalIP();
    console.log('✅ Server listening on:');
    console.log(`   http://localhost:${PORT}`);
    console.log(`   http://${ip}:${PORT}`);
    console.log('👉 Run `ngrok http ' + PORT + '` to expose this server publicly.');
  } catch (err) {
    console.error('❌ Startup error:', err?.message || err);
    process.exit(1);
  }
});
