export const sendResponse = (res, statusCode, message, data, errors) => {
  res.status(statusCode).json({
    success: statusCode >= 200 && statusCode < 300,
    message: message || 'Request was successful',
    data: data || null,
    errors: errors || null,
  });
};
