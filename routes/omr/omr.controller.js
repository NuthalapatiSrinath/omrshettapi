// controllers/omr.controller.js
import OMRModel from '../../database/modals/OMR.modal.js';
import { sendResponse } from '../../services/responseService.js';
import mongoose from 'mongoose';

export const fetchOMRSheets = async (req, res) => {
  const { author } = req.query;

  try {
    let omrRecords;

    if (author) {
      // return user + default
      omrRecords = await OMRModel.find({
        $or: [{ author }, { author: 'default' }],
      }).sort({ createdAt: -1 });
    } else {
      // return everything
      omrRecords = await OMRModel.find().sort({ createdAt: -1 });
    }

    console.log('[fetchOMRSheets] returning OMRs:', omrRecords.length);

    return sendResponse(
      res,
      200,
      'OMRs fetched successfully',
      omrRecords,
      null
    );
  } catch (error) {
    return sendResponse(
      res,
      500,
      'Error while fetching OMR sheets',
      null,
      error
    );
  }
};

export const createOMR = async (req, res) => {
  try {
    const {
      numberOfQuestions,
      numberOfOptions,
      numberOfIntegerQuestions = 0,
      author,
    } = req.body;

    const errors = {};
    if (!numberOfQuestions)
      errors.numberOfQuestions = 'Number of Questions is missing';
    if (!numberOfOptions)
      errors.numberOfOptions = 'Number of Options is missing';
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
      return sendResponse(
        res,
        409,
        'OMR with these specs already exists',
        null,
        null
      );
    }

    const newOMR = await OMRModel.create({
      numberOfQuestions,
      numberOfOptions,
      numberOfIntegerQuestions,
      author,
    });

    console.log('[createOMR] created:', newOMR);

    return sendResponse(
      res,
      201,
      'OMR sheet created successfully',
      newOMR,
      null
    );
  } catch (error) {
    if (error instanceof mongoose.Error.ValidationError) {
      const formattedErrors = Object.entries(error.errors).map(
        ([key, err]) => ({
          name: key,
          message: err.message,
        })
      );
      return sendResponse(res, 400, 'Validation Error', null, formattedErrors);
    }

    return sendResponse(res, 500, 'Error creating OMR', null, [
      { name: 'server', message: error.message },
    ]);
  }
};
