FROM node:20-slim AS base

# # Install pnpm
# ENV PNPM_HOME="/pnpm"
# ENV PATH="$PNPM_HOME:$PATH"
# RUN corepack enable

WORKDIR /app
COPY . /app

# FROM base AS prod-deps
# RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --prod --frozen-lockfile


# FROM base
FROM node:20-slim

ARG NEXT_PUBLIC_API_URL=http://localhost:3000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

WORKDIR /app
COPY . /app

RUN npm i
RUN npm run build
# RUN pnpm install --prod --frozen-lockfile
# RUN pnpm build

EXPOSE 8000
CMD [ "npm", "run", "start" ]
# CMD [ "pnpm", "start" ]


# FROM base
# COPY --from=prod-deps /app/node_modules /app/node_modules
# COPY --from=build /app/dist /app/dist
# EXPOSE 8000
# CMD [ "pnpm", "start" ]