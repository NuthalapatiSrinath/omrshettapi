import API from '..';

export const fetchOMRSheets = async (author) => {
  const response = await API.get(
    author ? `omr/fetchOMRsheets?author=${author}` : 'omr/fetchOMRsheets'
  );
  return response.data;
};

export const createOMR = async (
  numberOfQuestions,
  numberOfOptions,
  numberOfIntegerQuestions,
  author = 'default'
) => {
  const formData = {
    numberOfQuestions,
    numberOfOptions,
    numberOfIntegerQuestions,
    author,
  };
  const response = await API.post('/omr/createOMR', formData);
  return response.data;
};
