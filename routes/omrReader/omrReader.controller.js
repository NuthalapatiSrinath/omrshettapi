// routes/omrReader/omrReader.controller.js
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

const PY_BIN = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : 'python3');
// Use 'py' on Windows (common) or 'python' if installed to PATH. You can set env var PYTHON_BIN as needed.

export const extractOMR = async (req, res) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({ success: false, message: 'No file uploaded.' });
  }

  const imagePath = file.path; // multer saved path (relative)
  const pyPath = path.resolve('python_scripts', 'omr_reader.py'); // project-root/python_scripts/omr_reader.py

  // Defensive: ensure python script exists
  if (!fs.existsSync(pyPath)) {
    // cleanup uploaded file
    try { fs.unlinkSync(imagePath); } catch (e) {}
    return res.status(500).json({ success: false, message: 'OMR python script not found on server.' });
  }

  try {
    // spawn python process
    const py = spawn(PY_BIN, [pyPath, imagePath], { windowsHide: true });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (data) => { stdout += data.toString(); });
    py.stderr.on('data', (data) => { stderr += data.toString(); });

    py.on('error', (err) => {
      // spawn failed
      console.error('[extractOMR] spawn error:', err);
    });

    py.on('close', (code) => {
      // remove temp uploaded file
      try { fs.unlinkSync(imagePath); } catch (e) { /* ignore */ }

      if (code !== 0) {
        console.error('[extractOMR] python exit code:', code, 'stderr:', stderr);
        return res.status(500).json({
          success: false,
          message: 'OMR processing failed on server.',
          error: stderr || `python exited with code ${code}`
        });
      }

      // parse stdout -> JSON
      try {
        const parsed = JSON.parse(stdout);
        // forward parsed JSON as-is
        return res.status(200).json(parsed);
      } catch (err) {
        console.error('[extractOMR] JSON parse error', err, 'stdout:', stdout);
        return res.status(500).json({
          success: false,
          message: 'Invalid response from OMR script.',
          raw: stdout,
          error: err.message
        });
      }
    });
  } catch (err) {
    console.error('[extractOMR] server error', err);
    try { fs.unlinkSync(imagePath); } catch (e) {}
    return res.status(500).json({ success: false, message: 'Server error', error: err.message });
  }
};

export default extractOMR;
