import styles from './OmrBoxGrid.module.css';

export default function OmrBoxGrid({ boxes = 6 }) {
  return (
    <div className={styles.container}>
      {Array.from({ length: boxes }).map((_, idx) => (
        <div key={idx} className={styles.box}></div>
      ))}
    </div>
  );
}
