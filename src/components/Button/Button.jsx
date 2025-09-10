import React from "react";
import styles from "./Button.module.css";

const Button = ({ type = "primary", label, onClick }) => {
  return (
    <button
      className={type === "primary" ? styles.primary : styles.secondary}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

export default Button;
