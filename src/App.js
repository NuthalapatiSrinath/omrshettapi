import { useSelector } from 'react-redux';
import './App.css';
import RenderModal from './modals/RenderModal/RenderModal';
import AppRoutes from './routes/AppRoutes';
import { ToastContainer, Bounce } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
  const isModalOpen = useSelector((state) => state.modal.isOpen);
  return (
    <div className="App">
      <AppRoutes />
      {isModalOpen && <RenderModal />}
      <div className="ToastContainer">
        <ToastContainer
          stacked
          position="top-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick={true}
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="light"
          transition={Bounce}
        />
      </div>
    </div>
  );
}

export default App;
