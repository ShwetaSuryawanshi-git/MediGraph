// File: MediGraph/firebase/firebase.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/firebase/firebase.js

/**
 * Firebase Initialization Module
 * Initializes Firebase app, Auth, and Firestore services
 */

import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import firebaseConfig from './firebase.config';

// Initialize Firebase App
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication
export const auth = getAuth(app);

// Initialize Cloud Firestore
export const db = getFirestore(app);

export default app;
