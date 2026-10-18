-- Add auth columns to users
ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'local';
ALTER TABLE users ADD COLUMN google_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);

-- Create OTP table
CREATE TABLE IF NOT EXISTS email_verifications (
    email TEXT PRIMARY KEY,
    otp_code TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL
);

-- Drop invite codes table (no longer needed)
DROP TABLE IF EXISTS invite_codes;
CREATE TABLE IF NOT EXISTS invite_codes (
    code TEXT PRIMARY KEY,
    created_by INTEGER,
    used_by INTEGER,
    used_at TIMESTAMP
);
