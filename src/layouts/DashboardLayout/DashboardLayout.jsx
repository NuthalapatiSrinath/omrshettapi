import TopBar from '../../pages/TopBar/TopBar';
import styles from './DashboardLayout.module.css';

import { Outlet } from 'react-router';

function DashboardLayout() {
  return (
    <div className={styles.DashboardLayout}>
      <TopBar />

      <Outlet />
    </div>
  );
}

export default DashboardLayout;
