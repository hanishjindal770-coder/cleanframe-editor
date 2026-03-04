/**
 * CleanFrame Editor - Main Application
 * =====================================
 * A video editing application for removing unwanted objects using inpainting.
 * 
 * ETHICAL NOTICE: This tool is intended for legitimate use cases only:
 * - Removing unwanted objects in your own footage
 * - Privacy protection (license plates, faces, addresses)
 * - Cleaning up personal video projects
 * 
 * NOT intended for removing watermarks, logos, or DRM protection.
 */

import React, { useState, useCallback } from 'react';
import UploadPage from './components/UploadPage';
import Editor from './components/Editor';
import './App.css';

// API base URL - empty string means same origin (production)
const API_BASE_URL = '';

function App() {
    // Application state
    const [currentView, setCurrentView] = useState('upload'); // 'upload' | 'editor' | 'result'
    const [videoId, setVideoId] = useState(null);
    const [resultUrl, setResultUrl] = useState(null);

    /**
     * Handle successful video upload
     * Transitions from upload page to editor
     */
    const handleUploadSuccess = useCallback((uploadedVideoId) => {
        setVideoId(uploadedVideoId);
        setCurrentView('editor');
    }, []);

    /**
     * Handle successful video processing
     * Transitions from editor to result view
     */
    const handleProcessingComplete = useCallback((url) => {
        setResultUrl(url);
        setCurrentView('result');
    }, []);

    /**
     * Reset application to initial state
     * Returns user to upload page
     */
    const handleReset = useCallback(() => {
        setVideoId(null);
        setResultUrl(null);
        setCurrentView('upload');
    }, []);

    /**
     * Go back to editor from result view
     */
    const handleBackToEditor = useCallback(() => {
        setCurrentView('editor');
        setResultUrl(null);
    }, []);

    return (
        <div className="app">
            {/* Header */}
            <header className="app-header">
                <div className="header-content">
                    <div className="logo" onClick={handleReset}>
                        <div className="logo-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="2" />
                                <path d="M12 8V16M8 12H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        </div>
                        <span className="logo-text">
                            <span className="gradient-text">CleanFrame</span> Editor
                        </span>
                    </div>

                    <nav className="header-nav">
                        {currentView !== 'upload' && (
                            <button className="btn btn-secondary btn-sm" onClick={handleReset}>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M3 12H21M3 12L9 6M3 12L9 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                New Video
                            </button>
                        )}
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <main className="app-main">
                {currentView === 'upload' && (
                    <UploadPage
                        apiBaseUrl={API_BASE_URL}
                        onUploadSuccess={handleUploadSuccess}
                    />
                )}

                {currentView === 'editor' && videoId && (
                    <Editor
                        apiBaseUrl={API_BASE_URL}
                        videoId={videoId}
                        onProcessingComplete={handleProcessingComplete}
                        onBack={handleReset}
                    />
                )}

                {currentView === 'result' && resultUrl && (
                    <ResultView
                        apiBaseUrl={API_BASE_URL}
                        videoId={videoId}
                        resultUrl={resultUrl}
                        onBackToEditor={handleBackToEditor}
                        onNewVideo={handleReset}
                    />
                )}
            </main>

            {/* Footer */}
            <footer className="app-footer">
                <p>
                    CleanFrame Editor v1.0 — For legitimate video editing only.
                    <span className="text-muted"> Not for removing watermarks or copyrighted content.</span>
                </p>
            </footer>
        </div>
    );
}

/**
 * Result View Component
 * Displays the processed video and provides download option
 */
function ResultView({ apiBaseUrl, videoId, resultUrl, onBackToEditor, onNewVideo }) {
    const fullResultUrl = `${apiBaseUrl}${resultUrl}`;
    const downloadUrl = `${apiBaseUrl}/download/${videoId}`;

    return (
        <div className="result-view animate-fade-in">
            <div className="result-container">
                <div className="result-header">
                    <div className="result-icon success">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                            <path d="M8 12L11 15L16 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                    </div>
                    <h2>Video Processing Complete!</h2>
                    <p className="text-secondary">Your video has been successfully processed with inpainting applied.</p>
                </div>

                <div className="result-preview card">
                    <video
                        controls
                        autoPlay
                        loop
                        className="result-video"
                        src={fullResultUrl}
                    >
                        Your browser does not support video playback.
                    </video>
                </div>

                <div className="result-actions">
                    <a
                        href={downloadUrl}
                        className="btn btn-primary btn-lg"
                        download
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 3V16M12 16L7 11M12 16L17 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M3 20H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        Download Edited Video
                    </a>

                    <button className="btn btn-secondary btn-lg" onClick={onBackToEditor}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M11 19L4 12M4 12L11 5M4 12H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Back to Editor
                    </button>

                    <button className="btn btn-secondary btn-lg" onClick={onNewVideo}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Process New Video
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;
