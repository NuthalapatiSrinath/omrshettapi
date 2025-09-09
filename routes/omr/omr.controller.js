// omr.controller.js
import OMRModel from '../../database/modals/OMR.modal.js';
import { sendResponse } from '../../services/responseService.js';
import mongoose from 'mongoose';

/**
 * Flexible fetch — handles:
 *  - author as plain string
 *  - author stored as JSON-string (substring match)
 *  - author stored as object (author.email or author.id)
 *  - fallback: returns recent documents if nothing matches
 */
export const fetchOMRSheets = async (req, res, next) => {
  const { author } = req.query;

  try {
    // If client provided an author, attempt flexible matching
    if (author) {
      const docs = await OMRModel.find({
        $or: [
          { author }, // exact string match
          { author: 'default' }, // explicit default
          // if author stored as object (e.g. { id:..., email: ... })
          { 'author.id': author },
          { 'author._id': author },
          { 'author.email': author },
          // substring match - catches JSON-stringified author or other embedded cases
          { author: { $regex: author, $options: 'i' } },
        ],
      }).sort({ createdAt: -1 }).lean();

      if (docs && docs.length) return sendResponse(res, 200, 'OMRs fetched successfully', docs, null);
      // continue to fallback if none found
    }

    // Fallback: return recent documents so UI can show something while we debug
    const recent = await OMRModel.find({}).sort({ createdAt: -1 }).limit(50).lean();
    return sendResponse(res, 200, 'OMRs fetched (fallback recent)', recent, null);
  } catch (error) {
    return sendResponse(res, 500, 'Error while fetching OMR sheets', null, error);
  }
};

/**
 * Create OMR (unchanged behaviour)
 */
export const createOMR = async (req, res) => {
  try {
    const {
      numberOfQuestions,
      numberOfOptions,
      numberOfIntegerQuestions = 0,
      author,
    } = req.body;

    const errors = {};
    if (!numberOfQuestions) errors.numberOfQuestions = 'Number of Questions is missing';
    if (!numberOfOptions) errors.numberOfOptions = 'Number of Options is missing';
    if (!author) errors.author = 'Author is missing';

    if (Object.keys(errors).length) {
      return sendResponse(res, 400, 'Missing required fields', null, errors);
    }

    const exists = await OMRModel.findOne({
      author,
      numberOfQuestions,
      numberOfOptions,
      numberOfIntegerQuestions,
    });

    if (exists) {
      return sendResponse(res, 409, 'OMR with these specs already exists', null, null);
    }

    const newOMR = await OMRModel.create({
      numberOfQuestions,
      numberOfOptions,
      numberOfIntegerQuestions,
      author,
    });

    return sendResponse(res, 201, 'OMR sheet created successfully', newOMR, null);
  } catch (error) {
    if (error instanceof mongoose.Error.ValidationError) {
      const formattedErrors = Object.entries(error.errors).map(([key, err]) => ({
        name: key,
        message: err.message,
      }));
      return sendResponse(res, 400, 'Validation Error', null, formattedErrors);
    }

    return sendResponse(res, 500, 'Error creating OMR', null, [{ name: 'server', message: error.message }]);
  }
};

/**
 * Temporary debug endpoint - returns collection name, count and one sample doc
 * Mount at GET /omr/debug to inspect exactly how author is stored
 */
export const debugOMR = async (req, res) => {
  try {
    const collectionName = OMRModel.collection && OMRModel.collection.name;
    const count = await OMRModel.countDocuments().catch(() => null);
    const sample = await OMRModel.findOne({}).lean().catch(() => null);
    return sendResponse(res, 200, 'Debug info', { collectionName, count, sample }, null);
  } catch (err) {
    return sendResponse(res, 500, 'Debug error', null, err);
  }
};
