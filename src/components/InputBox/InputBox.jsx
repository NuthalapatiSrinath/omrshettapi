import React from 'react';
import styles from './InputBox.module.css';
import clsx from 'clsx';

function InputBox({
  label,
  type = 'text',
  className = '',
  value,
  onChange,
  placeholder,
  isError = false,
}) {
  const id = label.replace(/\s+/g, '-').toLowerCase();
  return (
    <div className={clsx(styles.InputBox, isError && styles.error)}>
      <label className={styles.label} htmlFor={id}>
        {label}
      </label>
      <input
        type={type}
        id={id}
        className={clsx(styles.input, className)}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

export default InputBox;
