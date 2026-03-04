import React, { useState, useRef } from 'react';
import './UploadPage.css';

function UploadPage({ apiBaseUrl, onUploadSuccess }) {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        validateAndSetFile(file);
    };

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        validateAndSetFile(file);
    };

    const validateAndSetFile = (file) => {
        setError(null);
        if (!file) return;

        const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            setError('Please select a valid video file (MP4, MOV, AVI, WebM)');
            return;
        }

        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            setError('File size must be under 500MB');
            return;
        }

        setSelectedFile(file);
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setUploading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(`${apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Upload failed');
            }

            const data = await response.json();
            setUploadProgress(100);

            setTimeout(() => {
                onUploadSuccess(data.video_id);
            }, 500);

        } catch (err) {
            setError(err.message || 'Failed to upload video');
            setUploading(false);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div className="upload-page">
            <div className="upload-container animate-slide-up">
                <div className="upload-header">
                    <h1>Remove Objects from Videos</h1>
                    <p>Upload a video and select areas to remove using AI-powered inpainting</p>
                </div>

                <div
                    className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !selectedFile && fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        onChange={handleFileSelect}
                        hidden
                    />

                    {!selectedFile ? (
                        <div className="dropzone-content">
                            <div className="dropzone-icon">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                                    <path d="M12 15V3M12 3L7 8M12 3L17 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                    <path d="M3 15V17C3 18.6569 4.34315 20 6 20H18C19.6569 20 21 18.6569 21 17V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                            </div>
                            <p className="dropzone-text">Drag & drop your video here</p>
                            <p className="dropzone-hint">or click to browse</p>
                            <p className="dropzone-formats">MP4, MOV, AVI, WebM • Max 500MB</p>
                        </div>
                    ) : (
                        <div className="file-preview">
                            <div className="file-icon">
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                                    <rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="2" />
                                    <path d="M10 9L15 12L10 15V9Z" fill="currentColor" />
                                </svg>
                            </div>
                            <div className="file-info">
                                <p className="file-name">{selectedFile.name}</p>
                                <p className="file-size">{formatFileSize(selectedFile.size)}</p>
                            </div>
                            <button className="btn btn-icon" onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                    <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                            </button>
                        </div>
                    )}
                </div>

                {error && <div className="toast toast-error">{error}</div>}

                {uploading && (
                    <div className="upload-progress">
                        <div className="progress-bar">
                            <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }}></div>
                        </div>
                        <p>Uploading... {uploadProgress}%</p>
                    </div>
                )}

                <button
                    className="btn btn-primary btn-lg w-full"
                    onClick={handleUpload}
                    disabled={!selectedFile || uploading}
                >
                    {uploading ? 'Uploading...' : 'Upload & Continue'}
                </button>

                <div className="upload-features">
                    <div className="feature">
                        <div className="feature-icon">🎯</div>
                        <h4>Select & Mask</h4>
                        <p>Draw over areas you want to remove</p>
                    </div>
                    <div className="feature">
                        <div className="feature-icon">🔄</div>
                        <h4>Track Objects</h4>
                        <p>Propagate masks across frames</p>
                    </div>
                    <div className="feature">
                        <div className="feature-icon">✨</div>
                        <h4>AI Inpaint</h4>
                        <p>Seamlessly fill removed areas</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UploadPage;
