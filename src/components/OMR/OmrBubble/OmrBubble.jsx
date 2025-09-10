import styles from './OmrBubble.module.css';

const OmrBubble = ({ value = 'A' }) => {
  return <div className={styles.OmrBubble}>{value}</div>;
};

export default OmrBubble;
