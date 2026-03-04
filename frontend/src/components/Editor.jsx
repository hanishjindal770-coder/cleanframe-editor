import React, { useState, useEffect, useRef, useCallback } from 'react';
import './Editor.css';

function Editor({ apiBaseUrl, videoId, onProcessingComplete, onBack }) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [videoInfo, setVideoInfo] = useState(null);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [masks, setMasks] = useState([]); // Array of mask objects with frame ranges
    const [isDrawing, setIsDrawing] = useState(false);
    const [drawStart, setDrawStart] = useState(null);
    const [currentRect, setCurrentRect] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [autoDetecting, setAutoDetecting] = useState(false);
    const [autoProcessing, setAutoProcessing] = useState(false);

    // New state for frame range selection
    const [pendingMask, setPendingMask] = useState(null); // Mask waiting for frame range
    const [startFrame, setStartFrame] = useState(0);
    const [endFrame, setEndFrame] = useState(0);

    const canvasRef = useRef(null);
    const imageRef = useRef(null);
    const containerRef = useRef(null);

    // Load frames on mount
    useEffect(() => {
        loadFrames();
    }, [videoId]);

    const loadFrames = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${apiBaseUrl}/frames/${videoId}`);
            if (!response.ok) throw new Error('Failed to load frames');
            const data = await response.json();
            setVideoInfo(data);
            setStartFrame(0);
            setEndFrame(data.total_frames - 1);
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Check if current frame is within any mask's range
    const getMasksForCurrentFrame = useCallback(() => {
        return masks.filter(mask =>
            currentFrameIndex >= mask.startFrame && currentFrameIndex <= mask.endFrame
        );
    }, [masks, currentFrameIndex]);

    // Draw the current frame and mask overlays
    const drawCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        const image = imageRef.current;
        if (!canvas || !image || !image.complete) return;

        const ctx = canvas.getContext('2d');
        const container = containerRef.current;
        if (!container) return;

        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight - 60;
        const scale = Math.min(containerWidth / image.naturalWidth, containerHeight / image.naturalHeight, 1);

        canvas.width = image.naturalWidth * scale;
        canvas.height = image.naturalHeight * scale;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

        // Draw existing masks that apply to current frame
        const activeMasks = getMasksForCurrentFrame();
        ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
        ctx.lineWidth = 2;

        activeMasks.forEach(mask => {
            const scaledRect = {
                x: mask.region.x * scale,
                y: mask.region.y * scale,
                width: mask.region.width * scale,
                height: mask.region.height * scale
            };
            ctx.fillRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
            ctx.strokeRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
        });

        // Draw pending mask preview
        if (pendingMask) {
            ctx.fillStyle = 'rgba(245, 158, 11, 0.4)';
            ctx.strokeStyle = 'rgba(245, 158, 11, 1)';
            ctx.setLineDash([5, 5]);
            const scaledRect = {
                x: pendingMask.x * scale,
                y: pendingMask.y * scale,
                width: pendingMask.width * scale,
                height: pendingMask.height * scale
            };
            ctx.fillRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
            ctx.strokeRect(scaledRect.x, scaledRect.y, scaledRect.width, scaledRect.height);
            ctx.setLineDash([]);
        }

        // Draw current selection
        if (currentRect) {
            ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
            ctx.strokeStyle = 'rgba(99, 102, 241, 1)';
            ctx.setLineDash([5, 5]);
            ctx.fillRect(currentRect.x, currentRect.y, currentRect.width, currentRect.height);
            ctx.strokeRect(currentRect.x, currentRect.y, currentRect.width, currentRect.height);
        }
    }, [currentFrameIndex, masks, currentRect, pendingMask, getMasksForCurrentFrame]);

    useEffect(() => {
        if (videoInfo && videoInfo.frame_urls.length > 0) {
            const img = new Image();
            img.onload = () => {
                imageRef.current = img;
                drawCanvas();
            };
            img.src = `${apiBaseUrl}${videoInfo.frame_urls[currentFrameIndex]}`;
        }
    }, [videoInfo, currentFrameIndex, apiBaseUrl, drawCanvas]);

    useEffect(() => {
        drawCanvas();
    }, [drawCanvas]);

    const getCanvasCoords = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    };

    const handleMouseDown = (e) => {
        if (pendingMask) return; // Don't allow new drawing while confirming a mask
        const coords = getCanvasCoords(e);
        setIsDrawing(true);
        setDrawStart(coords);
        setCurrentRect({ x: coords.x, y: coords.y, width: 0, height: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing || !drawStart) return;
        const coords = getCanvasCoords(e);
        setCurrentRect({
            x: Math.min(drawStart.x, coords.x),
            y: Math.min(drawStart.y, coords.y),
            width: Math.abs(coords.x - drawStart.x),
            height: Math.abs(coords.y - drawStart.y)
        });
    };

    const handleMouseUp = () => {
        if (!isDrawing || !currentRect || currentRect.width < 5 || currentRect.height < 5) {
            setIsDrawing(false);
            setDrawStart(null);
            setCurrentRect(null);
            return;
        }

        const canvas = canvasRef.current;
        const image = imageRef.current;
        if (!canvas || !image) return;

        const scale = canvas.width / image.naturalWidth;
        const originalRect = {
            x: Math.round(currentRect.x / scale),
            y: Math.round(currentRect.y / scale),
            width: Math.round(currentRect.width / scale),
            height: Math.round(currentRect.height / scale)
        };

        // Set as pending mask for frame range selection
        setPendingMask(originalRect);
        setStartFrame(currentFrameIndex);
        setEndFrame(Math.min(currentFrameIndex + 30, (videoInfo?.total_frames || 1) - 1));

        setIsDrawing(false);
        setDrawStart(null);
        setCurrentRect(null);
    };

    const confirmMask = () => {
        if (!pendingMask) return;

        const newMask = {
            id: Date.now(),
            region: pendingMask,
            startFrame: Math.min(startFrame, endFrame),
            endFrame: Math.max(startFrame, endFrame)
        };

        setMasks(prev => [...prev, newMask]);
        setPendingMask(null);
    };

    const cancelPendingMask = () => {
        setPendingMask(null);
    };

    const removeMask = (maskId) => {
        setMasks(prev => prev.filter(m => m.id !== maskId));
    };

    const clearAllMasks = () => {
        setMasks([]);
        setPendingMask(null);
    };

    const handleProcess = async () => {
        if (masks.length === 0) {
            setError('Please add at least one mask before processing');
            return;
        }

        try {
            setProcessing(true);
            setError(null);

            const response = await fetch(`${apiBaseUrl}/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: videoId,
                    masks: masks.map(m => ({
                        region: m.region,
                        start_frame: m.startFrame,
                        end_frame: m.endFrame
                    }))
                })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Processing failed');
            }

            const data = await response.json();
            onProcessingComplete(data.result_url);
        } catch (err) {
            setError(err.message);
        } finally {
            setProcessing(false);
        }
    };

    /**
     * Auto-detect text/watermarks in the video and add them as masks
     */
    const handleAutoDetect = async () => {
        try {
            setAutoDetecting(true);
            setError(null);

            const response = await fetch(`${apiBaseUrl}/auto-detect/${videoId}?sample_interval=20`, {
                method: 'POST'
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Auto-detection failed');
            }

            const data = await response.json();

            if (data.detections.length === 0) {
                setError('No text or watermarks detected in the video');
                return;
            }

            // Group similar detections and create masks
            const groupedDetections = groupSimilarDetections(data.detections);
            const newMasks = groupedDetections.map((group, index) => ({
                id: Date.now() + index,
                region: group.region,
                startFrame: group.startFrame,
                endFrame: group.endFrame
            }));

            setMasks(prev => [...prev, ...newMasks]);
        } catch (err) {
            setError(err.message);
        } finally {
            setAutoDetecting(false);
        }
    };

    /**
     * Group similar detections (same position) into single masks with frame ranges
     */
    const groupSimilarDetections = (detections) => {
        const groups = [];
        const threshold = 30; // pixels tolerance for matching

        for (const detection of detections) {
            const existingGroup = groups.find(g =>
                Math.abs(g.region.x - detection.region.x) < threshold &&
                Math.abs(g.region.y - detection.region.y) < threshold &&
                Math.abs(g.region.width - detection.region.width) < threshold * 2 &&
                Math.abs(g.region.height - detection.region.height) < threshold * 2
            );

            if (existingGroup) {
                // Expand the region to cover both and extend frame range
                existingGroup.region = {
                    x: Math.min(existingGroup.region.x, detection.region.x),
                    y: Math.min(existingGroup.region.y, detection.region.y),
                    width: Math.max(
                        existingGroup.region.x + existingGroup.region.width,
                        detection.region.x + detection.region.width
                    ) - Math.min(existingGroup.region.x, detection.region.x),
                    height: Math.max(
                        existingGroup.region.y + existingGroup.region.height,
                        detection.region.y + detection.region.height
                    ) - Math.min(existingGroup.region.y, detection.region.y)
                };
                existingGroup.startFrame = Math.min(existingGroup.startFrame, detection.frame_index);
                existingGroup.endFrame = Math.max(existingGroup.endFrame, detection.frame_index);
            } else {
                groups.push({
                    region: detection.region,
                    startFrame: detection.frame_index,
                    endFrame: Math.min(detection.frame_index + 30, (videoInfo?.total_frames || 1) - 1)
                });
            }
        }

        // Extend each group's frame range to cover the full video if it's likely a watermark
        return groups.map(g => ({
            ...g,
            startFrame: 0,
            endFrame: (videoInfo?.total_frames || 1) - 1
        }));
    };

    /**
     * One-click automatic text/watermark detection and removal
     */
    const handleAutoProcess = async () => {
        try {
            setAutoProcessing(true);
            setError(null);

            const response = await fetch(`${apiBaseUrl}/auto-process/${videoId}?min_confidence=0.4&sample_interval=20`, {
                method: 'POST'
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Auto-processing failed');
            }

            const data = await response.json();

            if (!data.result_url) {
                setError(data.message || 'No text detected to remove');
                return;
            }

            onProcessingComplete(data.result_url);
        } catch (err) {
            setError(err.message);
        } finally {
            setAutoProcessing(false);
        }
    };

    const formatTime = (frameIndex) => {
        if (!videoInfo) return '0:00';
        const seconds = frameIndex / videoInfo.fps;
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    if (loading) {
        return (
            <div className="editor-loading">
                <div className="spinner"></div>
                <p>Loading video frames...</p>
            </div>
        );
    }

    if (error && !videoInfo) {
        return (
            <div className="editor-error">
                <p>{error}</p>
                <button className="btn btn-primary" onClick={onBack}>Go Back</button>
            </div>
        );
    }

    return (
        <div className="editor">
            <div className="editor-sidebar">
                <div className="sidebar-section">
                    <h3>Timeline</h3>
                    <div className="timeline-slider">
                        <input
                            type="range"
                            min="0"
                            max={(videoInfo?.total_frames || 1) - 1}
                            value={currentFrameIndex}
                            onChange={(e) => setCurrentFrameIndex(parseInt(e.target.value))}
                        />
                        <div className="timeline-info">
                            <span>Frame {currentFrameIndex + 1} / {videoInfo?.total_frames || 0}</span>
                            <span className="time-display">{formatTime(currentFrameIndex)}</span>
                        </div>
                    </div>
                </div>

                {/* Frame Range Selection for Pending Mask */}
                {pendingMask && (
                    <div className="sidebar-section pending-mask-section">
                        <h3>🎯 Set Frame Range</h3>
                        <p className="hint">Specify which frames to apply this mask</p>

                        <div className="frame-range-inputs">
                            <div className="range-input-group">
                                <label>Start Frame</label>
                                <div className="input-with-time">
                                    <input
                                        type="number"
                                        min="0"
                                        max={(videoInfo?.total_frames || 1) - 1}
                                        value={startFrame}
                                        onChange={(e) => setStartFrame(parseInt(e.target.value) || 0)}
                                    />
                                    <span className="time-badge">{formatTime(startFrame)}</span>
                                </div>
                            </div>

                            <div className="range-input-group">
                                <label>End Frame</label>
                                <div className="input-with-time">
                                    <input
                                        type="number"
                                        min="0"
                                        max={(videoInfo?.total_frames || 1) - 1}
                                        value={endFrame}
                                        onChange={(e) => setEndFrame(parseInt(e.target.value) || 0)}
                                    />
                                    <span className="time-badge">{formatTime(endFrame)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="range-preview">
                            <span>Affects {Math.abs(endFrame - startFrame) + 1} frames</span>
                            <span>({formatTime(Math.min(startFrame, endFrame))} - {formatTime(Math.max(startFrame, endFrame))})</span>
                        </div>

                        <div className="pending-actions">
                            <button className="btn btn-primary" onClick={confirmMask}>
                                ✓ Confirm Mask
                            </button>
                            <button className="btn btn-secondary" onClick={cancelPendingMask}>
                                ✕ Cancel
                            </button>
                        </div>
                    </div>
                )}

                <div className="sidebar-section">
                    <h3>Masks ({masks.length})</h3>
                    <div className="mask-list">
                        {masks.map((mask) => (
                            <div key={mask.id} className="mask-item">
                                <div className="mask-info">
                                    <span className="mask-frames">
                                        Frame {mask.startFrame + 1} → {mask.endFrame + 1}
                                    </span>
                                    <span className="mask-time">
                                        {formatTime(mask.startFrame)} - {formatTime(mask.endFrame)}
                                    </span>
                                </div>
                                <button
                                    className="remove-btn"
                                    onClick={() => removeMask(mask.id)}
                                    title="Remove mask"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                        {masks.length === 0 && (
                            <p className="no-masks">No masks added yet. Draw on the frame to create one.</p>
                        )}
                    </div>
                    {masks.length > 0 && (
                        <button className="btn btn-secondary btn-sm w-full" onClick={clearAllMasks}>
                            Clear All Masks
                        </button>
                    )}
                </div>

                {/* Auto Detection Section */}
                <div className="sidebar-section auto-detect-section">
                    <h3>✨ Auto Detection</h3>
                    <p className="hint">Automatically detect and remove text/watermarks</p>

                    <button
                        className="btn btn-accent btn-md w-full"
                        onClick={handleAutoDetect}
                        disabled={autoDetecting || autoProcessing || processing}
                    >
                        {autoDetecting ? (
                            <>
                                <span className="spinner-small"></span>
                                Scanning...
                            </>
                        ) : (
                            <>
                                🔍 Detect Watermarks
                            </>
                        )}
                    </button>

                    <div className="divider-or">
                        <span>or</span>
                    </div>

                    <button
                        className="btn btn-gradient btn-md w-full"
                        onClick={handleAutoProcess}
                        disabled={autoDetecting || autoProcessing || processing}
                    >
                        {autoProcessing ? (
                            <>
                                <span className="spinner-small"></span>
                                Auto Removing...
                            </>
                        ) : (
                            <>
                                🚀 One-Click Remove All
                            </>
                        )}
                    </button>
                    <p className="hint-small">Detects & removes all text in one step</p>
                </div>

                <div className="sidebar-actions">
                    <button
                        className="btn btn-primary btn-lg w-full"
                        onClick={handleProcess}
                        disabled={processing || masks.length === 0 || autoProcessing}
                    >
                        {processing ? 'Processing...' : `Process Video (${masks.length} masks)`}
                    </button>
                </div>
            </div>

            <div className="editor-main" ref={containerRef}>
                <div className="canvas-container">
                    <canvas
                        ref={canvasRef}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                        style={{ cursor: pendingMask ? 'not-allowed' : 'crosshair' }}
                    />
                    <p className="canvas-hint">
                        {pendingMask
                            ? '👆 Set the frame range in the sidebar, then confirm the mask'
                            : 'Click and drag to select the area you want to remove'}
                    </p>
                </div>
                {error && <div className="toast toast-error">{error}</div>}
            </div>

            {processing && (
                <div className="processing-overlay">
                    <div className="processing-modal">
                        <div className="spinner"></div>
                        <h3>Processing Video</h3>
                        <p>Applying inpainting to {masks.length} mask region(s)...</p>
                    </div>
                </div>
            )}

            {autoProcessing && (
                <div className="processing-overlay">
                    <div className="processing-modal auto-process-modal">
                        <div className="spinner"></div>
                        <h3>🚀 Auto Removing Watermarks</h3>
                        <p>Detecting and removing all text/watermarks automatically...</p>
                        <div className="process-steps">
                            <span className="step active">1. Scanning frames</span>
                            <span className="step">2. Detecting text</span>
                            <span className="step">3. Removing regions</span>
                            <span className="step">4. Reassembling video</span>
                        </div>
                    </div>
                </div>
            )}

            {autoDetecting && (
                <div className="processing-overlay">
                    <div className="processing-modal">
                        <div className="spinner"></div>
                        <h3>🔍 Scanning for Watermarks</h3>
                        <p>Analyzing video frames for text and watermarks...</p>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Editor;
