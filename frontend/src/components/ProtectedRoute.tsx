import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export const ProtectedRoute: React.FC = () => {
  const { token, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center font-mono text-sm text-[#8b8b8b]">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_10px_#fff] animate-pulse" />
          <span>Loading session…</span>
        </div>
      </div>
    );
  }

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
};

export default ProtectedRoute;
