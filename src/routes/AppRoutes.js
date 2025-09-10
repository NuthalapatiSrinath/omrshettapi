import { Routes, Route } from 'react-router';
import DashboardLayout from '../layouts/DashboardLayout/DashboardLayout';
import Dashboard from '../pages/Dashboard/Dashboard';
import LoginPage from '../pages/LoginPage/LoginPage';
import OmrContainer from '../pages/OmrContainer/OmrContainer';
import UploadOmr from '../pages/UploadOmr/UploadOmr'; // 👈 added
import Results from '../pages/Results/Results';

function AppRoutes() {
  return (
    <Routes>
      {/* Standalone Routes */}
      <Route path="/test" element={<div>Welcome to the App</div>} />
      <Route path="login" element={<LoginPage />} />
      <Route path="/omrSheet" element={<OmrContainer />} />
      <Route path="/printOmrSheet" element={<OmrContainer />} />

      {/* Routes inside dashboard layout */}
      <Route path="/" element={<DashboardLayout />}>
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="uploadOmr" element={<UploadOmr />} />
        <Route path="/results" element={<Results />} />
 {/* 👈 added */}
      </Route>
    </Routes>
  );
}

export default AppRoutes;
