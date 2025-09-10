import React from "react";
import styles from "./IconAndName.module.css";
import { LuListTodo } from "react-icons/lu"; // icon shown in your screenshot

const IconAndName = ({
  title = "Generated OMR's",
  onClick,
  textColor = "var(--Primary_Color)",
}) => {
  return (
    <div className={styles.container} onClick={onClick}>
      <div className={styles.iconCircle}>
        <LuListTodo size={20} color="var(--Primary_Color)" />
      </div>
      <span className={styles.label} style={{ color: textColor }}>
        {title}
      </span>
    </div>
  );
};

export default IconAndName;
