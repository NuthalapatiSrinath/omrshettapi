// src/components/TopBar/TopBar.jsx
import React from "react";
import styles from "./TopBar.module.css";
import LogoIcon from "../../assets/images/Logo.png"; 
import Photo from "../../assets/images/Photo.png"; 
import { NavLink } from "react-router-dom";

function TopBar() {
  return (
    <div className={styles.topbar}>
      <div className={styles.leftSection}>
        <img src={LogoIcon} alt="Logo" className={styles.logo} />
        <h1 className={styles.title}>Digi Akshara</h1>
      </div>

      <div className={styles.centerSection}>
        <NavLink
          to="/dashboard"
          className={({ isActive }) => (isActive ? styles.active : styles.link)}
        >
          OMR Sheets
        </NavLink>
        <NavLink
          to="/exams"
          className={({ isActive }) => (isActive ? styles.active : styles.link)}
        >
          Exams
        </NavLink>
        <NavLink
          to="/results"
          className={({ isActive }) => (isActive ? styles.active : styles.link)}
        >
          Get Results
        </NavLink>
      </div>

      <div className={styles.rightSection}>
        <img src={Photo} alt="Admin" className={styles.avatar} />
        <span className={styles.adminLabel}>Admin</span>
      </div>
    </div>
  );
}

export default TopBar;
