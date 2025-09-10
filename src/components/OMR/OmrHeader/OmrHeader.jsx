import styles from './OmrHeader.module.css';

import RollNumberSection from '../RollNumberSection/RollNumberSection';
import DetailsContainer from '../DetailsContainer/DetailsContainer';
import SignatureContainer from '../SignatureContainer/SignatureContainer';
import OmrInstructions from '../OmrInstructions/OmrInstructions';
import TestBookletCode from '../TestBookletCode/TestBookletCode';

export default function OmrHeader() {
  return (
    <div className={styles.OmrHeaderContainer}>
      <RollNumberSection digits={9} />
      <div className={styles.rightSection}>
        <div className={styles.rightTopSection}>
          <DetailsContainer />
          <div className={styles.SignaturesBox}>
            <SignatureContainer label={'Candidate Sign'} />
            <SignatureContainer label={'Invigilator Sign'} />
          </div>
        </div>
        <div className={styles.rightBottomSection}>
          <OmrInstructions />
          <TestBookletCode />
        </div>
      </div>
    </div>
  );
}
