import styles from './OmrInstructions.module.css';
import WrongMethodsImage from './../../../assets/images/wrongMethodBubbling.png';
import CorrectMethodsImage from './../../../assets/images/correctMethodBubbling.png';

export default function OmrInstructions() {
  return (
    <div className={styles.OmrInstructions}>
      <span className={styles.title}>Instructions</span>
      <div className={styles.InstructionsBox}>
        <span>1.Use only blue/ black ball point pen to fill the circles.</span>
        <span>2.Do not crush or fold this sheet</span>
        <span>3.Do not make any stray marks on this sheet</span>
        <span>4.Circles should be darkened completely and properly.</span>
        <span>5.Use of pencil is strictly prohibited.</span>
      </div>

      <div className={styles.methods}>
        <div className={styles.Method}>
          <span>WRONG METHODS</span>
          <img src={WrongMethodsImage} />
        </div>
        <div className={styles.Method}>
          <span>CORRECT METHOD</span>
          <img src={CorrectMethodsImage} />
        </div>
      </div>
    </div>
  );
}
