import styles from './SignatureContainer.module.css';

export default function SignatureContainer({ label = 'Signature' }) {
  return (
    <div className={styles.container}>
      <div className={styles.label}>{label}</div>
    </div>
  );
}
