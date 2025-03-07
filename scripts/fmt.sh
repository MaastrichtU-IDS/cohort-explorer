#!/bin/bash

cd backend
hatch run fmt

cd ../frontend
npm run fmt
