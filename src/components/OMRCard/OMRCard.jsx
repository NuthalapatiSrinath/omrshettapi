import React from 'react';
import styles from './OMRCard.module.css';
import Button from '../Button/Button';

const OMRCard = ({
  imageSrc,
  intQuesCount,
  mcqCount,
  optionsCount,
  onView,
  onDownload,
}) => {
  return (
    <div className={styles.card}>
      <img src={imageSrc} alt="OMR Sheet" className={styles.image} />
      <p className={styles.label}>
        {mcqCount} MCQ's : {optionsCount} Options
      </p>
      <p className={styles.label}>{intQuesCount} Integer Questions</p>
      <div className={styles.buttonGroup}>
        <Button type="primary" label="View" onClick={onView} />
        <Button type="secondary" label="Download" onClick={onDownload} />
      </div>
    </div>
  );
};

export default OMRCard;
