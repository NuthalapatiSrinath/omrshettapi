import styles from './OmrSheet.module.css';
import OmrHeader from './../../components/OMR/OmrHeader/OmrHeader';
import McqOptions from '../../components/OMR/McqOptions/McqOptions';
import IntegerOmr from '../../components/OMR/IntegerOmr/IntegerOmr';

export default function OmrSheet({
  numberOfQuestions,
  numberOfOptions,
  numberOfIntegerQuestions,
}) {
  return (
    <div className={styles.OmrSheet}>
      <div className={styles.OMRheader}>
        <OmrHeader />
      </div>
      <div className={styles.OMRbody}>
        <div className={styles.QuestionsContainer}>
          {Array.from({ length: numberOfQuestions }, (_, i) => (
            <McqOptions
              key={`mcq-${i}`}
              questionNumber={i + 1}
              numberOfOptions={numberOfOptions}
            />
          ))}
        </div>

        <div className={styles.QuestionsContainer}>
          {Array.from({ length: numberOfIntegerQuestions }, (_, i) => (
            <IntegerOmr
              key={`int-${i}`}
              questionNumber={numberOfQuestions + i + 1}
              digitCount={6}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
