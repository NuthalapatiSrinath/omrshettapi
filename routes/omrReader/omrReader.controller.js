import express from 'express';
import multer from 'multer';
import dotenv from 'dotenv';
import fs from 'fs';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import {
  TextractClient,
  DetectDocumentTextCommand,
} from '@aws-sdk/client-textract';

dotenv.config();
const router = express.Router();

// Multer setup
const upload = multer({ dest: 'uploads/' });

// AWS S3 setup
const s3Client = new S3Client({
  region: process.env.S3_BUCKET_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY,
    secretAccessKey: process.env.SECRET_ACCESS_KEY,
  },
});

// AWS Textract setup
const textractClient = new TextractClient({
  region: process.env.S3_BUCKET_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY,
    secretAccessKey: process.env.SECRET_ACCESS_KEY,
  },
});

// Route: POST /readOMR (with file)
export const extractOMR = async (req, res) => {
  const file = req.file;

  console.log(file);

  if (!file) {
    return res
      .status(400)
      .json({ success: false, message: 'No file uploaded.' });
  }

  const fileStream = fs.createReadStream(file.path);
  const s3Key = `uploads/${Date.now()}_${file.originalname}`;

  try {
    // 1. Upload to S3
    await s3Client.send(
      new PutObjectCommand({
        Bucket: process.env.S3_BUCKET_NAME,
        Key: s3Key,
        Body: fileStream,
        ContentType: file.mimetype,
      })
    );

    // 2. Remove temp file
    fs.unlinkSync(file.path);

    // 3. Call Textract
    const textractResponse = await textractClient.send(
      new DetectDocumentTextCommand({
        Document: {
          S3Object: {
            Bucket: process.env.S3_BUCKET_NAME,
            Name: s3Key,
          },
        },
      })
    );

    const lines = textractResponse.Blocks.filter(
      (block) => block.BlockType === 'LINE'
    ).map((block) => block.Text);

    return res.status(200).json({
      success: true,
      extractedText: lines,
    });
  } catch (err) {
    console.error('Error:', err);
    return res.status(500).json({
      success: false,
      message: 'Text extraction failed.',
      error: err.message,
    });
  }
};

export default router;
