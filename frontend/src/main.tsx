import React from 'react';
import ReactDOM from 'react-dom/client';
import Home from './pages/Home';
import PoseEditor from './pages/PoseEditor';
import './index.css';

const isPoseEditor = window.location.pathname === '/pose-editor';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isPoseEditor ? <PoseEditor /> : <Home />}
  </React.StrictMode>
);
