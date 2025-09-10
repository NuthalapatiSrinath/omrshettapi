import styles from "./ProfileMenu.module.css";
import Photo from "./../../assets/images/Photo.png";
import { IoIosArrowDown } from "react-icons/io";

function ProfileMenu() {
  return (
    <div className={styles.ProfileMenu}>
      <div className={styles.ProfileMenuAvatar}>
        <img src={Photo} alt="User Avatar" />
      </div>
      <div className={styles.ProfileMenuInfo}>
        
        <p>Admin</p>
      </div>
      
    </div>
  );
}

export default ProfileMenu;
