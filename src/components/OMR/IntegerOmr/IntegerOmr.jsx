import styles from './IntegerOmr.module.css';
import OmrBubble from '../OmrBubble/OmrBubble';
import OmrBoxGrid from '../OmrBoxGrid/OmrBoxGrid';

export default function IntegerOmr({ questionNumber = 1, digitCount = 6 }) {
  const digits = ['/', '.', ...Array.from({ length: 10 }, (_, i) => String(i))];

  return (
    <div className={styles.integerOmr}>
      <span className={styles.questionNumber}>{questionNumber}</span>
      <div className={styles.columns}>
        <OmrBoxGrid boxes={digitCount} />
        <div className={styles.bubbleGrid}>
          {Array.from({ length: digitCount }).map((_, colIdx) => (
            <div key={colIdx} className={styles.bubbleColumn}>
              {digits.map((digit, rowIdx) => (
                <OmrBubble key={rowIdx} value={digit} />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
