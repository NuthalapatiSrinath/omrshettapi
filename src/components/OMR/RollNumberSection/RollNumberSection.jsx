import styles from './RollNumberSection.module.css';
import OmrBoxGrid from './../OmrBoxGrid/OmrBoxGrid';
import OmrBubble from './../OmrBubble/OmrBubble';

export default function RollNumberSection({ digits = 6 }) {
  const rows = Array.from({ length: 10 }, (_, i) => i); // 0–9
  const columns = Array.from({ length: digits });

  return (
    <div className={styles.wrapper}>
      <div className={styles.label}>Admission No :</div>

      {/* Boxes on top */}
      <OmrBoxGrid boxes={digits} />

      {/* Bubbles grid */}
      <div className={styles.bubbleGrid}>
        {rows.map((digit) => (
          <div key={digit} className={styles.row}>
            {columns.map((_, colIdx) => (
              <OmrBubble key={colIdx} value={digit} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
