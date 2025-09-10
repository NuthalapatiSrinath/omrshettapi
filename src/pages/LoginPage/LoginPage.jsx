import React, { useState } from "react";
import styles from "./LoginPage.module.css";
import loginIllustration from "../../assets/images/login.png";
import { FaUser, FaLock, FaEnvelope } from "react-icons/fa";
import logo from "../../assets/images/Logo.png";

const LoginPage = () => {
  const [formData, setFormData] = useState({
    role: "",
    username: "",
    password: "",
  });

  const handleChange = (field) => (e) => {
    setFormData((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Login Data:", formData);
  };

  return (
    <div className={styles.container}>
      <div className={styles.left}>
        <div className={styles.loginBox}>
          <img src={logo} alt="Logo" className={styles.logo} />
          <h2 className={styles.heading}>SIGN IN TO YOUR ACCOUNT</h2>

          <form onSubmit={handleSubmit} className={styles.form}>
            <label>Role</label>
            <div className={styles.inputWrapper}>
              <FaUser  />
              <select
                value={formData.role}
                onChange={handleChange("role")}
                required
              >
                <option value="">Select Role</option>
                <option value="admin">Admin</option>
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
              </select>
            </div>

            <label>Username</label>
            <div className={styles.inputWrapper}>
              <FaEnvelope />
              <input
                type="text"
                placeholder="Enter username"
                value={formData.username}
                onChange={handleChange("username")}
                required
              />
            </div>

            <label>Password</label>
            <div className={styles.inputWrapper}>
              <FaLock />
              <input
                type="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange("password")}
                required
              />
            </div>

            <div className={styles.actions}>
              <span className={styles.forgot}>Forgot Password ?</span>
              <button type="submit" className={styles.signInBtn}>
                Sign in
              </button>
            </div>
          </form>
        </div>
      </div>

      <div className={styles.right}>
        <div className={styles.curvedBackground}>
          <img src={loginIllustration} alt="login visual" />
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
