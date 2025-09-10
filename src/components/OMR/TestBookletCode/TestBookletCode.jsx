import OmrBubble from '../OmrBubble/OmrBubble';
import styles from './TestBookletCode.module.css';

export default function TestBookletCode({ options = ['A', 'B', 'C', 'D'] }) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.title}>Test Booklet Code</div>
      <div className={styles.options}>
        {options.map((value, idx) => (
          <OmrBubble key={idx} value={value} />
        ))}
      </div>
      <div className={styles.note}>
        Before Handing Over The Answer Sheet To The Invigilator, The Candidate
        Should Check That Test Booklet Code And Roll No. Have Been Filled And
        Marked Correctly.
      </div>
    </div>
  );
}
