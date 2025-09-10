import { Schema, model } from 'mongoose';

const OMRSchema = new Schema(
  {
    numberOfQuestions: {
      type: Number,
      required: true,
      min: 0,
    },
    numberOfOptions: {
      type: Number,
      required: true,
      min: 2,
      max: 6,
    },
    numberOfIntegerQuestions: {
      type: Number,
      default: 0,
      min: 0,
    },
    author: {
      type: String,
      required: true,
      trim: true,
      index: true,
    },
  },
  {
    timestamps: true,
    toJSON: {
      transform(doc, ret) {
        delete ret.updatedAt, delete ret.createdAt, delete ret.__v;
        return ret;
      },
    },
  }
);

const OMRModel = model('OMRSheet', OMRSchema);

export default OMRModel;
