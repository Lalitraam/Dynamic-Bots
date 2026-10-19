import React, { createContext, useContext, useState, useEffect } from 'react';
import apiClient from '../api/client';

export interface User {
  id: number;
  email: string;
  role: string;
  created_at?: string | null;
}

interface AuthContextType {
  token: string | null;
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<{ success: boolean; message: string; dev_otp?: string }>;
  sendOtp: (email: string) => Promise<{ success: boolean; message: string; dev_otp?: string }>;
  verifyEmail: (email: string, otp_code: string) => Promise<{ success: boolean }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem('token') || localStorage.getItem('dg_token')
  );
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchMe = async () => {
      const storedToken = localStorage.getItem('token') || localStorage.getItem('dg_token');
      if (storedToken) {
        try {
          const res = await apiClient.get<User>('/api/auth/me', {
            headers: { Authorization: `Bearer ${storedToken}` },
          });
          setUser(res.data);
          setToken(storedToken);
        } catch {
          localStorage.removeItem('token');
          localStorage.removeItem('dg_token');
          setToken(null);
          setUser(null);
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    };

    fetchMe();
  }, []);

  const login = async (email: string, password: string): Promise<void> => {
    const res = await apiClient.post<{ access_token: string }>('/api/auth/login', {
      email,
      password,
    });
    const access_token = res.data.access_token;
    localStorage.setItem('token', access_token);
    localStorage.setItem('dg_token', access_token);
    setToken(access_token);

    // Fetch user profile
    const meRes = await apiClient.get<User>('/api/auth/me', {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    setUser(meRes.data);
  };

  const register = async (
    email: string,
    password: string
  ): Promise<{ success: boolean; message: string; dev_otp?: string }> => {
    const res = await apiClient.post<{ message: string; dev_otp?: string }>('/api/auth/register', {
      email,
      password,
    });
    return {
      success: true,
      message: res.data.message || 'Verification code sent to your email.',
      dev_otp: res.data.dev_otp,
    };
  };

  const sendOtp = async (
    email: string
  ): Promise<{ success: boolean; message: string; dev_otp?: string }> => {
    const res = await apiClient.post<{ message: string; dev_otp?: string }>('/api/auth/send-otp', {
      email,
    });
    return {
      success: true,
      message: res.data.message || 'Verification code sent to your email.',
      dev_otp: res.data.dev_otp,
    };
  };

  const verifyEmail = async (email: string, otp_code: string): Promise<{ success: boolean }> => {
    const res = await apiClient.post<{ access_token: string }>('/api/auth/verify-email', {
      email,
      otp_code,
    });
    const access_token = res.data.access_token;
    localStorage.setItem('token', access_token);
    localStorage.setItem('dg_token', access_token);
    setToken(access_token);

    // Fetch user profile
    const meRes = await apiClient.get<User>('/api/auth/me', {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    setUser(meRes.data);

    return { success: true };
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('dg_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ token, user, loading, login, register, sendOtp, verifyEmail, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
