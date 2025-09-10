import React from "react";
import styles from "./SearchBar.module.css";
import { IoSearchOutline } from "react-icons/io5";

function SearchBar({ placeholder = "Search here...", onChange }) {
  return (
    <div className={styles.SearchBar}>
      <IoSearchOutline style={{ color: "blue" }} size={24} />
      <input
        type="search"
        placeholder={placeholder}
        onChange={onChange}
        className={styles.input}
      />
    </div>
  );
}

export default SearchBar;
