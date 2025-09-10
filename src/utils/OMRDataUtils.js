export const filterUserGeneratedOMR = (data, author) => {
  return data.filter((item) => {
    return item.author === author;
  });
};
export const filterDefaultOMR = (data) => {
  return data.filter((item) => {
    return item.author === 'default';
  });
};
