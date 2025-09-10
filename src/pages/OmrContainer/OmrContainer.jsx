import { useEffect } from 'react';
import OmrSheet from '../OmrSheet/OmrSheet';
import styles from './OmrContainer.module.css';
import { useLocation, useNavigate } from 'react-router-dom';

function OmrContainer() {
  const { state, pathname } = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (pathname === '/printOmrSheet') {
      // Wait for the component to fully render before printing
      const printTimeout = setTimeout(() => {
        window.print();
      }, 500);

      // Listen for the afterprint event to go back
      const handleAfterPrint = () => {
        navigate(-1); // Go back to previous page
      };

      window.addEventListener('afterprint', handleAfterPrint);

      return () => {
        clearTimeout(printTimeout);
        window.removeEventListener('afterprint', handleAfterPrint);
      };
    }
  }, [pathname, navigate]);

  return (
    <div className={styles.OmrContainer}>
      <div>
        <OmrSheet
          numberOfQuestions={state?.numberOfQuestions}
          numberOfOptions={state?.numberOfOptions}
          numberOfIntegerQuestions={state?.numberOfIntegerQuestions}
        />
      </div>
    </div>
  );
}

export default OmrContainer;
