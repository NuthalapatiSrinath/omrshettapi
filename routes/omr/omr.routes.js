import { Router } from 'express';
import { createOMR, fetchOMRSheets, debugOMR } from './omr.controller.js';
const router = Router();

router.post('/createOMR', createOMR);
router.get('/fetchOMRSheets', fetchOMRSheets); // ensure exact capitalization
router.get('/debug', debugOMR); // temporary

export default router;
