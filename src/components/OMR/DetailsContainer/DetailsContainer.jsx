import styles from './DetailsContainer.module.css';

export default function DetailsContainer({ label = 'Name' }) {
  return (
    <div className={styles.container}>
      <div className={styles.label}>{`${label} :`}</div>
    </div>
  );
}
