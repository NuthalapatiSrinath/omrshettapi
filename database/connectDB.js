// ./database/connectDB.js
import mongoose from 'mongoose';

const DEFAULT_RETRY_ATTEMPTS = 5;
const DEFAULT_RETRY_DELAY_MS = 2000; // base for exponential backoff

/**
 * Connect to MongoDB with retries and graceful handling.
 * Reads process.env.MONGO_URI
 */
async function connectDB({
  uri = process.env.MONGO_URI,
  attempts = DEFAULT_RETRY_ATTEMPTS,
  delayMs = DEFAULT_RETRY_DELAY_MS
} = {}) {
  if (!uri) {
    throw new Error('MONGO_URI is not provided in environment variables.');
  }

  // optional debug
  if (process.env.MONGO_DEBUG === 'true') {
    mongoose.set('debug', true);
  }

  // Recommended mongoose options (modern)
  const options = {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    // autoIndex: false, // consider disabling in production for performance
  };

  let lastErr = null;
  for (let i = 1; i <= attempts; i++) {
    try {
      await mongoose.connect(uri, options);
      console.log('✅ Connected to MongoDB');
      // attach event listeners for helpful logs
      mongoose.connection.on('connected', () => {
        console.log('Mongoose connection: connected');
      });
      mongoose.connection.on('reconnected', () => {
        console.log('Mongoose connection: reconnected');
      });
      mongoose.connection.on('disconnected', () => {
        console.warn('Mongoose connection: disconnected');
      });
      mongoose.connection.on('error', (err) => {
        console.error('Mongoose connection error:', err);
      });

      // handle graceful shutdown
      const gracefulExit = async () => {
        console.log('Closing MongoDB connection due to app termination');
        try {
          await mongoose.disconnect();
          console.log('MongoDB disconnected gracefully');
          process.exit(0);
        } catch (err) {
          console.error('Error during mongoose.disconnect()', err);
          process.exit(1);
        }
      };
      process.on('SIGINT', gracefulExit).on('SIGTERM', gracefulExit).on('exit', gracefulExit);

      return mongoose; // success
    } catch (err) {
      lastErr = err;
      const wait = delayMs * Math.pow(2, i - 1);
      console.warn(`MongoDB connection attempt ${i} failed. Retrying in ${wait}ms...`, err.message);
      await new Promise((r) => setTimeout(r, wait));
    }
  }

  console.error('Failed to connect to MongoDB after retries:', lastErr);
  throw lastErr;
}

// export default connectDB
export default connectDB;
