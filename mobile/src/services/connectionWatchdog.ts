/**
 * SmartVisionAI - Connection Watchdog Service (Phase 13)
 *
 * Responsibilities:
 * - Deterministic connection health tracking across live camera frame requests
 * - State machine:
 *     CONNECTED  : Normal real-time pipeline operation
 *     DEGRADED   : 1st consecutive frame or API failure (transient glitch guard)
 *     LOST       : 2nd consecutive failure (immediate fail-safe trigger)
 *     RECOVERING : 1st successful health check following LOST
 *     CONNECTED  : 2nd consecutive successful health check (full recovery confirmation)
 * - Exponential backoff health polling:
 *     2s -> 4s -> 8s -> 16s -> 30s max
 * - Concurrency guard: Zero overlapping health check requests
 * - Pure state management: Watchdog does NOT directly speak, vibrate, or alter
 *   navigation outputs; it provides clean state inspection and observer subscriptions.
 */

import { getHealthUrl } from "src/constants/config";

/* =========================================================
   TYPES & INTERFACES
   ========================================================= */

export type ConnectionState = "CONNECTED" | "DEGRADED" | "LOST" | "RECOVERING";

export interface ConnectionContext {
  consecutiveFailures: number;
  consecutiveSuccesses: number;
  lastFailureTime?: number;
  lastSuccessTime?: number;
  error?: string;
  currentBackoffMs: number;
}

export type ConnectionStateListener = (
  state: ConnectionState,
  previousState: ConnectionState,
  context: ConnectionContext
) => void;

/* =========================================================
   CONSTANTS
   ========================================================= */

export const BACKOFF_STEPS_MS: readonly number[] = [
  2000,
  4000,
  8000,
  16000,
  30000,
] as const;

export const INITIAL_BACKOFF_MS = 2000;
export const MAX_BACKOFF_MS = 30000;
export const HEALTH_REQUEST_TIMEOUT_MS = 4000;

/* =========================================================
   CONNECTION WATCHDOG CLASS
   ========================================================= */

export class ConnectionWatchdog {
  private state: ConnectionState = "CONNECTED";
  private consecutiveFailures: number = 0;
  private consecutiveSuccesses: number = 0;
  private lastFailureTime: number = 0;
  private lastSuccessTime: number = 0;
  private lastError: string = "";

  private backoffIndex: number = 0;
  private isCheckingHealth: boolean = false;
  private isRunning: boolean = false;
  private pollWhenConnected: boolean = false;
  private pollingTimer: ReturnType<typeof setTimeout> | null = null;
  private activeAbortController: AbortController | null = null;

  private readonly listeners: Set<ConnectionStateListener> = new Set();

  /**
   * Return current connection state.
   */
  public getState(): ConnectionState {
    return this.state;
  }

  /**
   * True if state is CONNECTED.
   */
  public isConnected(): boolean {
    return this.state === "CONNECTED";
  }

  /**
   * True if state is DEGRADED.
   */
  public isDegraded(): boolean {
    return this.state === "DEGRADED";
  }

  /**
   * True if state is LOST.
   */
  public isLost(): boolean {
    return this.state === "LOST";
  }

  /**
   * True if state is RECOVERING.
   */
  public isRecovering(): boolean {
    return this.state === "RECOVERING";
  }

  /**
   * Return current consecutive failure count.
   */
  public getConsecutiveFailures(): number {
    return this.consecutiveFailures;
  }

  /**
   * Return current consecutive success count.
   */
  public getConsecutiveSuccesses(): number {
    return this.consecutiveSuccesses;
  }

  /**
   * Return current exponential backoff duration in milliseconds.
   */
  public getCurrentBackoffMs(): number {
    return BACKOFF_STEPS_MS[this.backoffIndex] ?? MAX_BACKOFF_MS;
  }

  /**
   * Subscribe to state transition events.
   * Returns an unsubscribe function.
   */
  public subscribe(listener: ConnectionStateListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Record a successful frame request or health response.
   *
   * Transitions:
   * - If DEGRADED: restores immediately to CONNECTED (1 success).
   * - If LOST: transitions to RECOVERING on 1st success.
   * - If RECOVERING: transitions to CONNECTED on 2nd consecutive success.
   * - If CONNECTED: remains CONNECTED.
   */
  public recordSuccess(): ConnectionState {
    const previousState = this.state;
    this.consecutiveSuccesses += 1;
    this.consecutiveFailures = 0;
    this.lastSuccessTime = Date.now();
    this.resetBackoff();

    let nextState: ConnectionState = this.state;

    if (this.state === "DEGRADED") {
      nextState = "CONNECTED";
    } else if (this.state === "LOST") {
      nextState = "RECOVERING";
    } else if (this.state === "RECOVERING") {
      if (this.consecutiveSuccesses >= 2) {
        nextState = "CONNECTED";
      }
    }

    this.state = nextState;

    if (nextState !== previousState) {
      this.notifyListeners(previousState);
    }

    // If recovering, schedule prompt next probe to verify second consecutive success
    if (this.isRunning && this.state === "RECOVERING") {
      this.scheduleNextHealthCheck(2000);
    }

    return this.state;
  }

  /**
   * Record a failed frame request or health check failure.
   *
   * Transitions:
   * - 1st failure from CONNECTED -> DEGRADED
   * - 2nd consecutive failure (or failure from DEGRADED/RECOVERING) -> LOST
   * - Failures while already LOST advance exponential retry backoff.
   */
  public recordFailure(error?: string): ConnectionState {
    const previousState = this.state;
    this.consecutiveFailures += 1;
    this.consecutiveSuccesses = 0;
    this.lastFailureTime = Date.now();
    if (error) {
      this.lastError = error;
    }

    let nextState: ConnectionState = this.state;

    if (this.state === "CONNECTED") {
      nextState = "DEGRADED";
    } else if (this.state === "DEGRADED" || this.state === "RECOVERING") {
      nextState = "LOST";
    } else if (this.state === "LOST") {
      // Advance exponential backoff step for subsequent retries
      this.advanceBackoff();
    }

    if (this.consecutiveFailures >= 2) {
      nextState = "LOST";
    }

    this.state = nextState;

    if (nextState !== previousState) {
      this.notifyListeners(previousState, error);
    }

    // If lost or running background watchdog, schedule next probe with backoff
    if (this.isRunning || this.state === "LOST") {
      this.scheduleNextHealthCheck(this.getCurrentBackoffMs());
    }

    return this.state;
  }

  /**
   * Perform an isolated backend health check with strict concurrency lock.
   * Never allows overlapping requests.
   */
  public async checkHealth(): Promise<boolean> {
    if (this.isCheckingHealth) {
      return false;
    }

    this.isCheckingHealth = true;
    const controller = new AbortController();
    this.activeAbortController = controller;

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, HEALTH_REQUEST_TIMEOUT_MS);

    try {
      const url = getHealthUrl();
      const response = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        signal: controller.signal,
      });

      if (!response.ok) {
        this.recordFailure(`HTTP ${response.status}`);
        return false;
      }

      const rawData: unknown = await response.json();
      const isHealthy =
        typeof rawData === "object" &&
        rawData !== null &&
        "status" in rawData &&
        String((rawData as { status: unknown }).status).toLowerCase() === "healthy";

      if (isHealthy) {
        this.recordSuccess();
        return true;
      }

      this.recordFailure("Backend returned unhealthy status");
      return false;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Health request failed";
      this.recordFailure(message);
      return false;
    } finally {
      clearTimeout(timeoutId);
      this.activeAbortController = null;
      this.isCheckingHealth = false;
    }
  }

  /**
   * Start the watchdog background health monitoring loop.
   */
  public start(pollWhenConnected: boolean = false): void {
    this.isRunning = true;
    this.pollWhenConnected = pollWhenConnected;

    if (this.state === "LOST" || this.state === "RECOVERING" || this.pollWhenConnected) {
      this.scheduleNextHealthCheck(this.getCurrentBackoffMs());
    }
  }

  /**
   * Stop health polling loop and cancel active timers.
   */
  public stop(): void {
    this.isRunning = false;
    if (this.pollingTimer !== null) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
  }

  /**
   * Reset watchdog state to initial CONNECTED state.
   */
  public reset(): void {
    this.stop();
    this.state = "CONNECTED";
    this.consecutiveFailures = 0;
    this.consecutiveSuccesses = 0;
    this.lastFailureTime = 0;
    this.lastSuccessTime = 0;
    this.lastError = "";
    this.resetBackoff();
    this.isCheckingHealth = false;
  }

  /**
   * Destroy the instance, clearing all timers and listeners.
   */
  public destroy(): void {
    this.reset();
    this.listeners.clear();
  }

  /* =========================================================
     PRIVATE HELPERS
     ========================================================= */

  private advanceBackoff(): void {
    if (this.backoffIndex < BACKOFF_STEPS_MS.length - 1) {
      this.backoffIndex += 1;
    }
  }

  private resetBackoff(): void {
    this.backoffIndex = 0;
  }

  private scheduleNextHealthCheck(delayMs: number): void {
    if (this.pollingTimer !== null) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }

    this.pollingTimer = setTimeout(() => {
      this.pollingTimer = null;
      void this.checkHealth();
    }, delayMs);
  }

  private notifyListeners(previousState: ConnectionState, error?: string): void {
    const currentState = this.state;
    const context: ConnectionContext = {
      consecutiveFailures: this.consecutiveFailures,
      consecutiveSuccesses: this.consecutiveSuccesses,
      lastFailureTime: this.lastFailureTime || undefined,
      lastSuccessTime: this.lastSuccessTime || undefined,
      error: error ?? this.lastError ?? undefined,
      currentBackoffMs: this.getCurrentBackoffMs(),
    };

    this.listeners.forEach((listener) => {
      try {
        listener(currentState, previousState, context);
      } catch (err) {
        console.warn("[ConnectionWatchdog] Listener callback error:", err);
      }
    });
  }
}

/* =========================================================
   DEFAULT SINGLETON INSTANCE
   ========================================================= */

export const connectionWatchdog = new ConnectionWatchdog();
export default connectionWatchdog;

