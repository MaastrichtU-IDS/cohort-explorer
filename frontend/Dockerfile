FROM node:20-slim

ARG NEXT_PUBLIC_API_URL=http://localhost:3000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . ./
RUN npm run build

EXPOSE 8000
CMD [ "npm", "run", "start" ]
