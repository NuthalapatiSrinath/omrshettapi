import { useSelector } from 'react-redux';
import styles from './RenderModal.module.css';
import MainModal from '../MainModal/MainModal';
import GenerateOmr from '../GenerateOmr/GenerateOmr';
import { useEffect } from 'react';
function RenderModal() {
  const activeModal = useSelector((state) => state.modal.type);

  useEffect(() => {
    document.body.classList.add('noScroll');

    return () => {
      document.body.classList.remove('noScroll');
    };
  });

  const allModals = {
    generateOmr: <GenerateOmr />,
  };

  return (
    <MainModal>
      <div className={styles.RenderModal}>{allModals[activeModal]}</div>
    </MainModal>
  );
}

export default RenderModal;
