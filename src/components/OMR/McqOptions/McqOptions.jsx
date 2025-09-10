import OmrBubble from '../OmrBubble/OmrBubble';
import styles from './McqOptions.module.css';

export default function McqOptions({
  questionNumber = 1,
  numberOfOptions = 4,
}) {
  const options = Array.from({ length: numberOfOptions }, (_, i) =>
    String.fromCharCode(65 + i)
  );

  return (
    <div className={styles.McqOptions}>
      <span className={styles.questionNumber}>{questionNumber}</span>
      <div className={styles.options}>
        {options.map((opt, idx) => (
          <OmrBubble key={idx} value={opt} />
        ))}
      </div>
    </div>
  );
}
