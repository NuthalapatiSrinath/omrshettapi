import express from 'express';
import multer from 'multer';
import { extractOMR } from './omrReader.controller.js';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/readOMR', upload.single('omrSheet'), extractOMR);

export default router;
