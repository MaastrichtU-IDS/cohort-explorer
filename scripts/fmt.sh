#!/bin/bash

cd backend
hatch run fmt

cd frontend
pnpm fmt
