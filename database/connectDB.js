// database/connectDB.js
import mongoose from 'mongoose';

const connectDB = async () => {
  const uri = process.env.MONGO_URI;

  if (!uri || typeof uri !== 'string' || !uri.trim()) {
    // Clear message and exit (or throw so caller can handle)
    const msg = [
      'MONGO_URI environment variable is missing or empty.',
      'Make sure you have a .env file in the project root with MONGO_URI set,',
      'or that you export MONGO_URI in your shell before starting the server.',
      'Example: MONGO_URI="mongodb://localhost:27017/digiakshara" npm run dev'
    ].join(' ');
    console.error('MongoDB connection error :', msg);
    // exit to avoid undefined being passed to mongoose
    process.exit(1);
    // or: throw new Error(msg);
  }

  try {
    // mongoose.connect returns the default mongoose instance. You can pass options if needed.
    const conn = await mongoose.connect(uri, {
      // options below are optional depending on your mongoose version
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log('Database Connected Successfully');
    console.log(`MongoDB Connected at: ${conn.connection.host}`);
  } catch (error) {
    console.error(`MongoDB connection error : ${error.message}`);
    process.exit(1);
  }
};

export default connectDB;
