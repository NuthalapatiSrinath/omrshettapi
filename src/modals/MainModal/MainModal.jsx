import { useSelector } from "react-redux";
import styles from "./MainModal.module.css";

function MainModal({ children }) {
  const activeModal = useSelector((state) => state.modal.type);

  // List of fullscreen modal types
  const isFullScreen = ["studentSection", "staffSection"].includes(activeModal); // Add more as needed

  return <div className={styles.MainModal}>{children}</div>;
}

export default MainModal;
