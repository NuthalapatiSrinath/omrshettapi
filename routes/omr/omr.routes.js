import { Router } from 'express';
import { createOMR, fetchOMRSheets } from './omr.controller.js';

const router = Router();

router.post('/createOMR', createOMR);
router.get('/fetchOMRsheets', fetchOMRSheets);

export default router;
